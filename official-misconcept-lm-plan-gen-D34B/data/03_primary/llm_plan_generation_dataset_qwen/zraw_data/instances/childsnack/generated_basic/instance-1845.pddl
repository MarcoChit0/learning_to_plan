; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.9
; random seed: 345715

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child278 child403 - child
    bread273 bread464 - bread-portion
    content212 content483 - content-portion
    tray453 - tray
    table415 table93 table20 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray453 kitchen)
     (at_kitchen_bread bread273)
     (at_kitchen_bread bread464)
     (at_kitchen_content content212)
     (at_kitchen_content content483)
     (no_gluten_bread bread273)
     (no_gluten_content content212)
     (allergic_gluten child403)
     (not_allergic_gluten child278)
     (waiting child278 table93)
     (waiting child403 table20)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child278)
     (served child403)
    )
  )
)
