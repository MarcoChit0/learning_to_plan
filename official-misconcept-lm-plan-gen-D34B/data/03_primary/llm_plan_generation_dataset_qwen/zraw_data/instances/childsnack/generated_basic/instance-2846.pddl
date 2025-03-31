; child-snack task with 1 children and 0.23 gluten factor 
; constant factor of 1.7
; random seed: 832081

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child408 - child
    bread259 - bread-portion
    content64 - content-portion
    tray410 tray53 - tray
    table452 table418 table359 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray410 kitchen)
     (at tray53 kitchen)
     (at_kitchen_bread bread259)
     (at_kitchen_content content64)
     (not_allergic_gluten child408)
     (waiting child408 table452)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child408)
    )
  )
)
