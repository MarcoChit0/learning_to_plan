; child-snack task with 2 children and 0.41 gluten factor 
; constant factor of 1.9
; random seed: 556340

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child484 child66 - child
    bread162 bread420 - bread-portion
    content399 content352 - content-portion
    tray200 tray490 tray484 tray247 - tray
    table351 table201 table311 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray200 kitchen)
     (at tray490 kitchen)
     (at tray484 kitchen)
     (at tray247 kitchen)
     (at_kitchen_bread bread162)
     (at_kitchen_bread bread420)
     (at_kitchen_content content399)
     (at_kitchen_content content352)
     (not_allergic_gluten child484)
     (not_allergic_gluten child66)
     (waiting child484 table311)
     (waiting child66 table351)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child484)
     (served child66)
    )
  )
)
