; child-snack task with 1 children and 0.69 gluten factor 
; constant factor of 1.3
; random seed: 441640

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child18 - child
    bread201 - bread-portion
    content203 - content-portion
    tray101 tray225 tray210 tray248 - tray
    table408 table75 table446 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray101 kitchen)
     (at tray225 kitchen)
     (at tray210 kitchen)
     (at tray248 kitchen)
     (at_kitchen_bread bread201)
     (at_kitchen_content content203)
     (not_allergic_gluten child18)
     (waiting child18 table446)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child18)
    )
  )
)
