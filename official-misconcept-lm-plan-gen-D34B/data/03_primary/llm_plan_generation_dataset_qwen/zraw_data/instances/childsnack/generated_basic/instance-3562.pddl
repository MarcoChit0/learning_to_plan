; child-snack task with 3 children and 0.36 gluten factor 
; constant factor of 1.3
; random seed: 323088

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child81 child225 child128 - child
    bread231 bread47 bread69 - bread-portion
    content390 content328 content111 - content-portion
    tray468 tray210 tray193 - tray
    table51 table180 table331 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray468 kitchen)
     (at tray210 kitchen)
     (at tray193 kitchen)
     (at_kitchen_bread bread231)
     (at_kitchen_bread bread47)
     (at_kitchen_bread bread69)
     (at_kitchen_content content390)
     (at_kitchen_content content328)
     (at_kitchen_content content111)
     (no_gluten_bread bread47)
     (no_gluten_content content328)
     (allergic_gluten child128)
     (not_allergic_gluten child81)
     (not_allergic_gluten child225)
     (waiting child81 table331)
     (waiting child225 table331)
     (waiting child128 table51)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child81)
     (served child225)
     (served child128)
    )
  )
)
