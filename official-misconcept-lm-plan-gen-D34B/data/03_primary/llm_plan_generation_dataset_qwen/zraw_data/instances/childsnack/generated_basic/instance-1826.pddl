; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.9
; random seed: 535687

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child92 child434 - child
    bread44 bread352 - bread-portion
    content42 content241 - content-portion
    tray72 - tray
    table61 table309 table330 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray72 kitchen)
     (at_kitchen_bread bread44)
     (at_kitchen_bread bread352)
     (at_kitchen_content content42)
     (at_kitchen_content content241)
     (no_gluten_bread bread352)
     (no_gluten_content content241)
     (allergic_gluten child434)
     (not_allergic_gluten child92)
     (waiting child92 table61)
     (waiting child434 table309)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child92)
     (served child434)
    )
  )
)
