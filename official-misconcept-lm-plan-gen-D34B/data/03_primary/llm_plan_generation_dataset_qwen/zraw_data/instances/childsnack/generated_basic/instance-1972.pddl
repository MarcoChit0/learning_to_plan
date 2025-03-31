; child-snack task with 2 children and 0.64 gluten factor 
; constant factor of 1.3
; random seed: 149811

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child465 child74 - child
    bread210 bread13 - bread-portion
    content19 content11 - content-portion
    tray283 - tray
    table329 table147 table295 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray283 kitchen)
     (at_kitchen_bread bread210)
     (at_kitchen_bread bread13)
     (at_kitchen_content content19)
     (at_kitchen_content content11)
     (no_gluten_bread bread13)
     (no_gluten_content content19)
     (allergic_gluten child465)
     (not_allergic_gluten child74)
     (waiting child465 table147)
     (waiting child74 table147)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child465)
     (served child74)
    )
  )
)
