; child-snack task with 2 children and 0.28 gluten factor 
; constant factor of 1.7
; random seed: 492615

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child83 child147 - child
    bread394 bread70 - bread-portion
    content320 content200 - content-portion
    tray327 - tray
    table233 table144 table290 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray327 kitchen)
     (at_kitchen_bread bread394)
     (at_kitchen_bread bread70)
     (at_kitchen_content content320)
     (at_kitchen_content content200)
     (not_allergic_gluten child83)
     (not_allergic_gluten child147)
     (waiting child83 table144)
     (waiting child147 table233)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child83)
     (served child147)
    )
  )
)
