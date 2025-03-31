; child-snack task with 3 children and 0.72 gluten factor 
; constant factor of 1.9
; random seed: 30439

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child243 child110 child374 - child
    bread200 bread7 bread37 - bread-portion
    content146 content28 content41 - content-portion
    tray196 tray238 - tray
    table368 table299 table408 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray196 kitchen)
     (at tray238 kitchen)
     (at_kitchen_bread bread200)
     (at_kitchen_bread bread7)
     (at_kitchen_bread bread37)
     (at_kitchen_content content146)
     (at_kitchen_content content28)
     (at_kitchen_content content41)
     (no_gluten_bread bread7)
     (no_gluten_bread bread200)
     (no_gluten_content content28)
     (no_gluten_content content146)
     (allergic_gluten child110)
     (allergic_gluten child243)
     (not_allergic_gluten child374)
     (waiting child243 table408)
     (waiting child110 table368)
     (waiting child374 table408)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child243)
     (served child110)
     (served child374)
    )
  )
)
