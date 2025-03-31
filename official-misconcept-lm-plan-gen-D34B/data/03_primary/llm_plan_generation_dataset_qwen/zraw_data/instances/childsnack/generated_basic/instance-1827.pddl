; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.9
; random seed: 697490

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child108 child221 - child
    bread113 bread336 - bread-portion
    content253 content373 - content-portion
    tray12 - tray
    table464 table117 table32 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray12 kitchen)
     (at_kitchen_bread bread113)
     (at_kitchen_bread bread336)
     (at_kitchen_content content253)
     (at_kitchen_content content373)
     (no_gluten_bread bread336)
     (no_gluten_content content373)
     (allergic_gluten child221)
     (not_allergic_gluten child108)
     (waiting child108 table464)
     (waiting child221 table464)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child108)
     (served child221)
    )
  )
)
