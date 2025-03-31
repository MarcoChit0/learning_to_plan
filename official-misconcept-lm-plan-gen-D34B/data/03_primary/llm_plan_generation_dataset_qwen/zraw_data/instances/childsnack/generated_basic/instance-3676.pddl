; child-snack task with 2 children and 0.8 gluten factor 
; constant factor of 1.7
; random seed: 599139

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child168 child134 - child
    bread31 bread481 - bread-portion
    content241 content120 - content-portion
    tray295 tray223 - tray
    table41 table168 table202 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray295 kitchen)
     (at tray223 kitchen)
     (at_kitchen_bread bread31)
     (at_kitchen_bread bread481)
     (at_kitchen_content content241)
     (at_kitchen_content content120)
     (no_gluten_bread bread481)
     (no_gluten_content content241)
     (allergic_gluten child134)
     (not_allergic_gluten child168)
     (waiting child168 table202)
     (waiting child134 table202)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child168)
     (served child134)
    )
  )
)
