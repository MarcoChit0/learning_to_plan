; child-snack task with 1 children and 0.43 gluten factor 
; constant factor of 1.9
; random seed: 685665

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child84 - child
    bread124 - bread-portion
    content244 - content-portion
    tray358 tray252 tray12 tray78 - tray
    table367 table454 table363 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray358 kitchen)
     (at tray252 kitchen)
     (at tray12 kitchen)
     (at tray78 kitchen)
     (at_kitchen_bread bread124)
     (at_kitchen_content content244)
     (not_allergic_gluten child84)
     (waiting child84 table367)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child84)
    )
  )
)
