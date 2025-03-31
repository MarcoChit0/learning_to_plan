; child-snack task with 1 children and 0.48 gluten factor 
; constant factor of 1.7
; random seed: 737704

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child152 - child
    bread92 - bread-portion
    content37 - content-portion
    tray206 tray377 tray473 - tray
    table96 table374 table98 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray206 kitchen)
     (at tray377 kitchen)
     (at tray473 kitchen)
     (at_kitchen_bread bread92)
     (at_kitchen_content content37)
     (not_allergic_gluten child152)
     (waiting child152 table98)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child152)
    )
  )
)
