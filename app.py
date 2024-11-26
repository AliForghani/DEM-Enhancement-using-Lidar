import streamlit as st
import pandas as pd
import geopandas as gpd
import laspy
import numpy as np
import pdal
import json
import rasterio
import geopandas as gpd
import os
from pathlib import Path
from shapely.geometry import Point, MultiPoint
from rasterio.transform import from_origin
from scipy.interpolate import griddata
from scipy.spatial import KDTree
import shutil
from rasterio.merge import merge
from rasterio.mask import mask
import glob
import xarray as xr
from io import BytesIO
import shapely.geometry
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def gpkg_to_las(points_gdf,app_generated_id):

    x = points_gdf.geometry.x
    y = points_gdf.geometry.y
    z = points_gdf['z'].values 

    header = laspy.LasHeader()
    las = laspy.LasData(header)
    las.x = x
    las.y = y
    las.z = z
    las.write(os.path.join(st.session_state['this_bridge_output'],'filtered_points','filtered_points_%s.las'%app_generated_id))

def make_tif(tiff_resolution,app_generated_id):
    my_pipe={
        "pipeline": [
            {
            "type": "readers.las",
            "filename": os.path.join(st.session_state['this_bridge_output'],'filtered_points','filtered_points_%s.las'%app_generated_id), 
            "spatialreference": "EPSG:3857"  # specify the correct coordinate reference system
            },
            
            {
            "type": "writers.gdal",
            "filename": os.path.join(st.session_state['this_bridge_output'],'tif_files', 'dem_%sm_%s.tif'%(str(tiff_resolution),app_generated_id)),
            "dimension": "Z",
            "output_type": "idw", # or try  "mean",
            "resolution": tiff_resolution,
            "nodata": -999,
            "data_type": "float32",
            }
                ]
    }

    # Create a PDAL pipeline object
    pipeline = pdal.Pipeline(json.dumps(my_pipe))

    # Execute the pipeline
    pipeline.execute()



def las_to_gpkg(las_path, poly_geo):
    las=laspy.read(las_path)

    #make x,y coordinates
    x_y = np.vstack((np.array(las.x), np.array(las.y))).transpose()

    #convert the coordinates to a list of shapely Points
    las_points = list(MultiPoint(x_y).geoms)

    #put the points in a GeoDataFrame for a more standard syntax through Geopandas
    points_gdf= gpd.GeoDataFrame(geometry=las_points, crs="epsg:3857")

    #make sure the points are within the input polygon
    #points_gdf=points_gdf[points_gdf.geometry.within(poly_geo)]

    #add other required data into gdf...here only elevation
    z_values=np.array(las.z)
    points_gdf['z']=z_values

    return_values=np.array(las.return_number)
    points_gdf['return']=return_values

    class_values=np.array(las.classification)
    points_gdf['classification']=class_values

    number_of_returns = np.array(las.number_of_returns)
    points_gdf['number_of_returns'] = number_of_returns
    
    #note that pdal uses below scenarios of 'return number' and 'number_of_return' as described in documentauon below to identify, first, and last returns ...
    # There is no other way to do this classifcation task. Remember that x,y of different returns of a specifc pulse can be in different locations. so you should not 
    # expect to have multiple points at a exact x, y with different return numbers. 
    #  Also, 'point_source_id' are usually not reliable.  So, the only way to assign return numbers is by comparing return number and 'number of returns' as pdal is doing:
    #https://pdal.io/en/latest/stages/filters.returns.html 

    return points_gdf


def make_lidar_foorprints():
    str_hobu_footprints = (
        r"https://raw.githubusercontent.com/hobu/usgs-lidar/master/boundaries/boundaries.topojson"
    )
    gdf_entwine_footprints = gpd.read_file(str_hobu_footprints)
    gdf_entwine_footprints.set_crs("epsg:4326", inplace=True)  #it is geographic (lat-long degrees) commonly used for GPS for accurate locations...slower tpo process for visualizations
    #it is important to use 3857 because poly_wkt in process_lidar must be in that crs to properly apply "readers.ept" step.
    return gdf_entwine_footprints


def add_classification_names(classification_counts):
    lidar_classification_codes = {
        0: "Unclassified",
        1: "Unclassified",
        2: "Ground",
        3: "Low Vegetation",
        4: "Medium Vegetation",
        5: "High Vegetation",
        6: "Building",
        7: "Low Point (Noise)",
        8: "Reserved",
        9: "Water",
        10: "Rail",
        11: "Road Surface",
        12: "Reserved",
        13: "Wire - Guard (or Bridge)",
        14: "Wire - Conductor",
        15: "Transmission Tower",
        16: "Wire-Structure Connector",
        17: "Bridge Deck",
        18: "High Noise"
    }

    classification_counts['Description']=classification_counts['Classification Code'].map(lidar_classification_codes)
    classification_counts=classification_counts[['Classification Code','Description','app_generated_id','Percentage of Points (%)']]

    # classification_counts = classification_counts.style.set_properties(**{'text-align': 'center'})
    return classification_counts

def download_lidar_points(poly_geo,lidar_url,tif_crs,app_generated_id):
    poly_wkt=poly_geo.wkt
    my_pipe={
    "pipeline": [
        {
            "polygon": str(poly_wkt),
            "filename": lidar_url,
            "type": "readers.ept",
            "tag": "readdata"
        },
        {
        "type": "filters.returns",
        "groups": "last,only"  #last mean the last if when there are multiple returns..it does not include cases When there is only one return. so we need both "last" and "only" 
        },

        {
            "in_srs":'EPSG:3857',
            "out_srs": 'EPSG:%d'%tif_crs,
            "type": "filters.reprojection",
            "tag": "reprojected",
        },

        {
        "filename": os.path.join(st.session_state['this_bridge_output'] , 'original_points','all_last_return_points_%s.las'%app_generated_id),
        "inputs": ["reprojected"],
        "tag": "writerslas",
        "type": "writers.las"
        }
            ]
        }

    # Create a PDAL pipeline object
    pipeline = pdal.Pipeline(json.dumps(my_pipe))

    # Execute the pipeline
    pipeline.execute()


    #make a gpkg file from points
    points_gdf=las_to_gpkg(os.path.join(st.session_state['this_bridge_output'],'original_points', 'all_last_return_points_%s.las'%app_generated_id),poly_geo)
    points_gdf.to_file(os.path.join(st.session_state['this_bridge_output'],'original_points','all_last_return_points_%s.gpkg'%app_generated_id))


    classification_counts = points_gdf.groupby('classification').size().reset_index()
    classification_counts.columns = ['Classification Code', 'Point Count']

    classification_counts['Percentage of Points (%)']=round(100*classification_counts['Point Count']/len(points_gdf),2)
    classification_counts.loc[:,'app_generated_id']=app_generated_id
    classification_counts=add_classification_names(classification_counts)
    
    classification_counts.to_csv(os.path.join(st.session_state['this_bridge_output'],'classifications','classifications_%s.csv'%app_generated_id), index=False)

    return classification_counts


def enhance_regional_with_local(regional_tif_path,selected_classes_dict, output_tif_path,tiff_resolution):
    # Step 1: Open the regional TIFF as an xarray DataArray
    original_da = xr.open_dataarray(regional_tif_path, engine="rasterio")
    original_nodata=original_da.rio.nodata
    enhanced_da = original_da.copy()

    # for app_generated_id in st.session_state['app_generated_ids']: 
    for app_generated_id, _ in selected_classes_dict.items():
        local_tif_path=os.path.join(st.session_state['this_bridge_output'],'tif_files','dem_%sm_%s.tif'%(str(tiff_resolution),app_generated_id))  
        local_da = xr.open_dataarray(local_tif_path, engine="rasterio")

        if local_da.rio.crs != original_da.rio.crs:
            local_da = local_da.rio.reproject_match(original_da) 

        # Replace values in the regional DataArray with the local DataArray values at overlapping locations
        enhanced_da = enhanced_da.where(local_da.isnull(), other=local_da)


    # # Set nodata value to be consistent
    enhanced_da = enhanced_da.fillna(original_nodata)
    enhanced_da.rio.write_nodata(original_nodata, inplace=True)

    # if needed, Explicitly assign the CRS from the regional data to the final combined data
    # updated_regional_da.rio.write_crs(regional_da.rio.crs, inplace=True)

    # Step 6: Save the combined DataArray to a new TIFF file
    enhanced_da.rio.to_raster(output_tif_path)



def create_output_folder(uploaded_file):
    file_name = uploaded_file.name
    file_name = os.path.splitext(os.path.basename(file_name))[0]

    #make a parent output folder
    this_bridge_output = os.path.join(results_folder, file_name,)
    os.makedirs(this_bridge_output, exist_ok=True)

    # Create subfolders 'lidar_points' and 'classifications' inside the main folder
    original_points_output = os.path.join(this_bridge_output, 'original_points')
    classifications_output = os.path.join(this_bridge_output, 'classifications')
    filtered_points_output = os.path.join(this_bridge_output, 'filtered_points')
    tif_files_output = os.path.join(this_bridge_output, 'tif_files')

    os.makedirs(original_points_output, exist_ok=True)
    os.makedirs(classifications_output, exist_ok=True)
    os.makedirs(filtered_points_output, exist_ok=True)
    os.makedirs(tif_files_output, exist_ok=True)

    return this_bridge_output,file_name

def zip_file_function():
    folder_to_zip=st.session_state['this_bridge_output']
    zip_output = str(st.session_state['this_bridge_output'])
    shutil.make_archive(zip_output, 'zip', folder_to_zip)


    # Load the zip file as binary data
    with open("%s.zip"%zip_output, "rb") as f:
        zip_file = f.read()

    #Create a download button for the zip file
    st.download_button(
        label="Download ZIP File",
        data=zip_file,
        file_name="%s.zip"%str(os.path.basename(zip_output)),  
        mime="application/zip"
    )


def make_plot():
    # Paths to the DEM files
    regional_original = st.session_state['regional_file_bytes']
    regional_updated = os.path.join(st.session_state['this_bridge_output'], 'regional_updated.tif')

    if Path(regional_updated).exists():
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Original DEM", "Updated DEM"))

        # Plot original DEM in the first column
        plot_dem_with_boundary(fig, regional_original, col_index=1 , show_legend=True)
        
        # Plot updated DEM in the second column
        plot_dem_with_boundary(fig, regional_updated, col_index=2 , show_legend=False)

        # Update layout and legends
        fig.update_layout(
            title="",
            coloraxis_colorbar=dict(title="Elevtion (m)"),
            showlegend=True,
            legend=dict(
                title="",
                orientation="v",
                x=0.45,
                y=1,
                bgcolor="rgba(255,255,255,0.5)",
                bordercolor="Black",
                borderwidth=1
            ),
            margin=dict(t=25)  # Reduce top margin to remove extra space
        )
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)

        # Save as HTML and provide a link
        html_file_path = os.path.join(st.session_state['this_bridge_output'], 'plot.html')
        fig.write_html(html_file_path)

        # html_link = f'<a href="{html_file_path}" target="_blank">Open Plot in New Tab</a>'
        # st.markdown(html_link, unsafe_allow_html=True)
        st.plotly_chart(fig)
    else:
        st.write('Generate the enhanced DEM first and retry')


def plot_dem_with_boundary(fig, dem_file, col_index,show_legend):
    """Function to plot DEM with polygon boundary on a specified subplot column."""
    
    # Step 1: Read the TIFF file using rasterio
    with rasterio.open(dem_file) as src:
        # Read the first band of the raster
        band1 = src.read(1)
        # Get the raster transformation to align the polygon correctly
        transform = src.transform

        # Generate x and y coordinates in the real-world CRS
        xs, ys = np.meshgrid(
            np.arange(band1.shape[1]), np.arange(band1.shape[0])
        )
        xs, ys = rasterio.transform.xy(transform, ys, xs, offset="center")
        
        # Flatten xs and ys for Plotly plotting
        xs = np.array(xs).flatten()
        ys = np.array(ys).flatten()
        band1_flat = band1.flatten()

    
    # Step 3: Plot the raster data with Plotly
    fig.add_trace(go.Heatmap(
        x=xs,
        y=ys,
        z=band1_flat,
        colorscale='viridis',
        colorbar=dict(title="Elevation (m)"),
        hovertemplate="z: %{z:.2f}<extra></extra>"),
        row=1, col=col_index
    )
    
    # Read the domain and reproject the polygon to match the raster CRS
    gdf = gpd.read_file(os.path.join(st.session_state['this_bridge_output'], 'bridge_domain.gpkg'))
    gdf = gdf.to_crs(src.crs)

    #  Overlay the polygon boundaries
    for idx, geom in enumerate(gdf.geometry):
        if geom.is_empty or geom.exterior is None:
            continue  # Skip empty geometries or invalid polygons

        # Extract boundary coordinates for the current polygon
        x, y = geom.exterior.xy

        # Generate the convex hull from the polygon boundary
        convex_hull = np.array(
            shapely.geometry.MultiPoint(
                [xy for xy in zip(x, y)]
            ).convex_hull.exterior.coords
        )

        # Add the current polygon boundary to the figure
        fig.add_trace(go.Scatter(
            x=convex_hull[:, 0], 
            y=convex_hull[:, 1], 
            mode='lines', 
            line=dict(color='red', width=2),
            name=f'Bridges' if show_legend else None,
            showlegend=show_legend and idx == 0,  # Show legend only once
            legendgroup='Polygon Boundary'
        ), row=1, col=col_index)

# Streamlit app title
st.title("Enhance DEMs Using Lidar")

# File uploader for GPKG file
col1, col2 = st.columns(2)
with col1:
    uploaded_file = st.file_uploader("Specify Bridge domain (a GPKG file)", type="gpkg")
with col2:
    regional_file_upload = st.file_uploader("Optional: Upload regional DEM", type="tif")

# Initialize session state for tracking display state of classification counts
if 'classification_counts' not in st.session_state:
    st.session_state.classification_counts = None

if 'run_of_enhancer' not in st.session_state:
    st.session_state.run_of_enhancer = False

tif_crs = 3857
results_folder = "Results"

lidar_ftprint_gdf = make_lidar_foorprints()
lidar_ftprint_gdf.to_crs("epsg:%d"%tif_crs, inplace=True)

# Button to load the GPKG file and show classifications
if uploaded_file and st.button('Show Classifications!'):
    this_bridge_output,file_name=create_output_folder(uploaded_file)
    st.session_state['this_bridge_output'] = this_bridge_output

    # file_basename
    polys = gpd.read_file(uploaded_file, open_options=["IMMUTABLE=YES"])

    #assign a unique id, which is especially required when there are multiple records
    polys['app_generated_id']=[f"{file_name}_{i}" for i in np.arange(1,len(polys)+1)]
    st.session_state['app_generated_ids'] = polys['app_generated_id'].values.tolist()

    polys.to_file(os.path.join(st.session_state['this_bridge_output'],'bridge_domain.gpkg'))
    polys.to_crs(lidar_ftprint_gdf.crs, inplace=True)
    #if the gpkg file has a column called 'name' because also entiwine has a 'name' it makes error
    if 'name' in polys.columns:
        polys.rename(columns={'name':'name_modified'}, inplace=True)

    intersection = gpd.overlay(polys, lidar_ftprint_gdf, how='intersection')
    print(intersection)
    

    classification_summary=[]
    for i, row in intersection.iterrows():
        poly_geo, lidar_url, app_generated_id = row.geometry, row.url, row.app_generated_id
        classification_counts = download_lidar_points(poly_geo, lidar_url, tif_crs,app_generated_id)
        classification_summary.append(classification_counts)


    classification_summary=pd.concat(classification_summary)
        
    # Save classification_counts in session state
    st.session_state['classification_counts'] = classification_summary


# Check if classification_counts DataFrame was loaded previously and display it
if st.session_state['classification_counts'] is not None:
    # st.write(st.session_state.classification_counts)
    shown_classes_df=st.dataframe(st.session_state['classification_counts'],selection_mode = ["multi-row"], hide_index=True,on_select='rerun')    

    st.subheader("Select settings for making tif files:")

    selected_classes_df=st.session_state['classification_counts'].iloc[shown_classes_df['selection']['rows'],:]
    selected_classes_dict = selected_classes_df.groupby('app_generated_id')['Classification Code'].apply(list).to_dict()



    # st.write('Selected lidar classifications are: %s' % ",".join(selected_classes.astype(str)))
    st.write(f'Selected lidar classifications are: {selected_classes_dict}')


    # Create two columns
    col1, col2 = st.columns([2,3])  # Adjust the width ratio as needed

    # Place the label in the first column and the text input in the second column
    with col1:
        st.markdown("**Tiff resolution (m):**")  # Display the label in bold

    with col2:
        tiff_resolution = st.text_input("dummy", label_visibility="collapsed")


    col1, col2 = st.columns([1,1])


    # After displaying the DataFrame, create a new button
    with col1:
        if st.button('Make local tiff file!'):  
            if selected_classes_df.size>0 and tiff_resolution:
                
                for app_generated_id, this_classes in selected_classes_dict.items(): #st.session_state['app_generated_ids']:
                    points_gdf=gpd.read_file(os.path.join(st.session_state['this_bridge_output'],'original_points','all_last_return_points_%s.gpkg'%app_generated_id))

                    #keep only selected 
                    points_gdf=points_gdf[points_gdf['classification'].isin(this_classes)]

                    #make a las file for subsequent pdal pipeline
                    gpkg_to_las(points_gdf,app_generated_id)
                    make_tif(tiff_resolution,app_generated_id)

                    st.success("dem_%sm_%s.tif file was created!"%(str(tiff_resolution),app_generated_id))   
            else:
                error_message = ""
                if not tiff_resolution:
                    error_message += "Select resolution. "
                if selected_classes_df.size==0:
                    error_message += "Select at least one classification code."

                st.error(error_message)     
    with col2:
    
        if st.button('Enhance Regional DEM!'): 
            
            output_tif_path=os.path.join(st.session_state['this_bridge_output'],'regional_updated.tif') 

            #rasterio expects a file path or a file-like object opened in binary mode, but Streamlit's UploadedFile object doesn’t work directly with rasterio. 
            # To resolve this, we can convert the UploadedFile object into a BytesIO object, which rasterio can read as an in-memory file        
            if regional_file_upload is not None:
                regional_file_bytes = BytesIO(regional_file_upload.read())
                st.session_state['regional_file_bytes']=regional_file_bytes
                enhance_regional_with_local(regional_file_bytes,selected_classes_dict, output_tif_path,tiff_resolution)
                st.success("regional_updated.tif file was created") 
                st.session_state['run_of_enhancer']=True

            else:
                st.error('Upload a regional DEM and retry')
                st.session_state['run_of_enhancer']=False

    
    if st.button('Make plots!'): 
        if st.session_state['run_of_enhancer']:
            make_plot()
        else:
            st.error('Make sure to upload a regional DEM, enhance the regional DEM and retry')
        
    
    if st.button('Generate Zip file!'):
        zip_file_function() 




