; child-snack task with 1 children and 0.36 gluten factor 
; constant factor of 1.3
; random seed: 856128

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child211 - child
    bread278 - bread-portion
    content497 - content-portion
    tray12 tray263 tray367 - tray
    table0 table299 table103 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray12 kitchen)
     (at tray263 kitchen)
     (at tray367 kitchen)
     (at_kitchen_bread bread278)
     (at_kitchen_content content497)
     (not_allergic_gluten child211)
     (waiting child211 table0)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child211)
    )
  )
)
