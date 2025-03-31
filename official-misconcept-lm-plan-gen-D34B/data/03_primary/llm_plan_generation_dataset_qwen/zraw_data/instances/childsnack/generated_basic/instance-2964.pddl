; child-snack task with 1 children and 0.69 gluten factor 
; constant factor of 1.3
; random seed: 382573

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child111 - child
    bread479 - bread-portion
    content18 - content-portion
    tray425 tray326 tray387 tray6 - tray
    table237 table40 table374 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray425 kitchen)
     (at tray326 kitchen)
     (at tray387 kitchen)
     (at tray6 kitchen)
     (at_kitchen_bread bread479)
     (at_kitchen_content content18)
     (not_allergic_gluten child111)
     (waiting child111 table374)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child111)
    )
  )
)
