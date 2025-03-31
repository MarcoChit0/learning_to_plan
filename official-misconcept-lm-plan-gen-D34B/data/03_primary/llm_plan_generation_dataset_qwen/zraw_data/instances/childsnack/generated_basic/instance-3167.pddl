; child-snack task with 2 children and 0.48 gluten factor 
; constant factor of 1.9
; random seed: 804731

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child434 child367 - child
    bread52 bread88 - bread-portion
    content288 content386 - content-portion
    tray70 tray240 - tray
    table139 table218 table213 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray70 kitchen)
     (at tray240 kitchen)
     (at_kitchen_bread bread52)
     (at_kitchen_bread bread88)
     (at_kitchen_content content288)
     (at_kitchen_content content386)
     (not_allergic_gluten child434)
     (not_allergic_gluten child367)
     (waiting child434 table139)
     (waiting child367 table213)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child434)
     (served child367)
    )
  )
)
