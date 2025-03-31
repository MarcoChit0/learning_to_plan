; child-snack task with 2 children and 0.55 gluten factor 
; constant factor of 1.9
; random seed: 185760

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child124 child397 - child
    bread364 bread313 - bread-portion
    content142 content253 - content-portion
    tray393 tray475 tray13 tray331 - tray
    table205 table27 table453 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray393 kitchen)
     (at tray475 kitchen)
     (at tray13 kitchen)
     (at tray331 kitchen)
     (at_kitchen_bread bread364)
     (at_kitchen_bread bread313)
     (at_kitchen_content content142)
     (at_kitchen_content content253)
     (no_gluten_bread bread313)
     (no_gluten_content content142)
     (allergic_gluten child124)
     (not_allergic_gluten child397)
     (waiting child124 table27)
     (waiting child397 table205)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child124)
     (served child397)
    )
  )
)
