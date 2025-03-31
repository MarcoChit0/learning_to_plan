; child-snack task with 1 children and 0.77 gluten factor 
; constant factor of 1.9
; random seed: 859664

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child395 - child
    bread172 - bread-portion
    content340 - content-portion
    tray372 tray466 tray168 tray446 - tray
    table432 table199 table158 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray372 kitchen)
     (at tray466 kitchen)
     (at tray168 kitchen)
     (at tray446 kitchen)
     (at_kitchen_bread bread172)
     (at_kitchen_content content340)
     (not_allergic_gluten child395)
     (waiting child395 table158)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child395)
    )
  )
)
