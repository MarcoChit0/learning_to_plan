; child-snack task with 2 children and 0.49 gluten factor 
; constant factor of 1.7
; random seed: 378763

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child489 child401 - child
    bread385 bread412 - bread-portion
    content206 content54 - content-portion
    tray277 tray74 tray351 tray39 - tray
    table345 table132 table40 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray277 kitchen)
     (at tray74 kitchen)
     (at tray351 kitchen)
     (at tray39 kitchen)
     (at_kitchen_bread bread385)
     (at_kitchen_bread bread412)
     (at_kitchen_content content206)
     (at_kitchen_content content54)
     (not_allergic_gluten child489)
     (not_allergic_gluten child401)
     (waiting child489 table132)
     (waiting child401 table132)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child489)
     (served child401)
    )
  )
)
