; child-snack task with 2 children and 0.55 gluten factor 
; constant factor of 1.7
; random seed: 435141

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child14 child74 - child
    bread185 bread91 - bread-portion
    content117 content180 - content-portion
    tray167 - tray
    table213 table287 table252 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray167 kitchen)
     (at_kitchen_bread bread185)
     (at_kitchen_bread bread91)
     (at_kitchen_content content117)
     (at_kitchen_content content180)
     (no_gluten_bread bread185)
     (no_gluten_content content117)
     (allergic_gluten child74)
     (not_allergic_gluten child14)
     (waiting child14 table287)
     (waiting child74 table213)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child14)
     (served child74)
    )
  )
)
