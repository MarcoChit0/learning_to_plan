; child-snack task with 1 children and 0.62 gluten factor 
; constant factor of 1.5
; random seed: 456890

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child233 - child
    bread78 - bread-portion
    content329 - content-portion
    tray420 tray79 tray146 tray3 - tray
    table34 table400 table248 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray420 kitchen)
     (at tray79 kitchen)
     (at tray146 kitchen)
     (at tray3 kitchen)
     (at_kitchen_bread bread78)
     (at_kitchen_content content329)
     (not_allergic_gluten child233)
     (waiting child233 table400)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child233)
    )
  )
)
