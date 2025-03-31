; child-snack task with 1 children and 0.3 gluten factor 
; constant factor of 1.3
; random seed: 331761

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child298 - child
    bread230 - bread-portion
    content412 - content-portion
    tray206 tray297 tray409 - tray
    table185 table367 table157 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray206 kitchen)
     (at tray297 kitchen)
     (at tray409 kitchen)
     (at_kitchen_bread bread230)
     (at_kitchen_content content412)
     (not_allergic_gluten child298)
     (waiting child298 table367)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child298)
    )
  )
)
