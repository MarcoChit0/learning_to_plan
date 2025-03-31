; child-snack task with 1 children and 0.36 gluten factor 
; constant factor of 1.5
; random seed: 749847

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child219 - child
    bread41 - bread-portion
    content491 - content-portion
    tray367 - tray
    table362 table95 table61 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray367 kitchen)
     (at_kitchen_bread bread41)
     (at_kitchen_content content491)
     (not_allergic_gluten child219)
     (waiting child219 table95)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child219)
    )
  )
)
