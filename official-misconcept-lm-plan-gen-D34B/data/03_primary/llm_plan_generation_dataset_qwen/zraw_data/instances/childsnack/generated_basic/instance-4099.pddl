; child-snack task with 1 children and 0.79 gluten factor 
; constant factor of 1.9
; random seed: 374356

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child261 - child
    bread195 - bread-portion
    content478 - content-portion
    tray426 tray9 tray121 - tray
    table366 table457 table289 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray426 kitchen)
     (at tray9 kitchen)
     (at tray121 kitchen)
     (at_kitchen_bread bread195)
     (at_kitchen_content content478)
     (not_allergic_gluten child261)
     (waiting child261 table366)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child261)
    )
  )
)
