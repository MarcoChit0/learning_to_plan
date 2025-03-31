; child-snack task with 2 children and 0.55 gluten factor 
; constant factor of 1.9
; random seed: 516828

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child189 child360 - child
    bread198 bread84 - bread-portion
    content230 content248 - content-portion
    tray491 tray182 tray211 tray131 - tray
    table333 table198 table71 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray491 kitchen)
     (at tray182 kitchen)
     (at tray211 kitchen)
     (at tray131 kitchen)
     (at_kitchen_bread bread198)
     (at_kitchen_bread bread84)
     (at_kitchen_content content230)
     (at_kitchen_content content248)
     (no_gluten_bread bread84)
     (no_gluten_content content248)
     (allergic_gluten child189)
     (not_allergic_gluten child360)
     (waiting child189 table198)
     (waiting child360 table333)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child189)
     (served child360)
    )
  )
)
