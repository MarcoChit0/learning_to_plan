; child-snack task with 2 children and 0.75 gluten factor 
; constant factor of 1.7
; random seed: 579386

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child235 child60 - child
    bread58 bread149 - bread-portion
    content426 content52 - content-portion
    tray21 tray426 - tray
    table98 table268 table241 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray21 kitchen)
     (at tray426 kitchen)
     (at_kitchen_bread bread58)
     (at_kitchen_bread bread149)
     (at_kitchen_content content426)
     (at_kitchen_content content52)
     (no_gluten_bread bread58)
     (no_gluten_content content52)
     (allergic_gluten child60)
     (not_allergic_gluten child235)
     (waiting child235 table241)
     (waiting child60 table98)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child235)
     (served child60)
    )
  )
)
