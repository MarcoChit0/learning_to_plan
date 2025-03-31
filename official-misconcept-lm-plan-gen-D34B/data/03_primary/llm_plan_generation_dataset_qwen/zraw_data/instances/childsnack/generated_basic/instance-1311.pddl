; child-snack task with 2 children and 0.78 gluten factor 
; constant factor of 1.5
; random seed: 646388

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child294 child374 - child
    bread410 bread144 - bread-portion
    content175 content119 - content-portion
    tray382 tray489 tray439 tray376 - tray
    table141 table23 table354 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray382 kitchen)
     (at tray489 kitchen)
     (at tray439 kitchen)
     (at tray376 kitchen)
     (at_kitchen_bread bread410)
     (at_kitchen_bread bread144)
     (at_kitchen_content content175)
     (at_kitchen_content content119)
     (no_gluten_bread bread144)
     (no_gluten_content content175)
     (allergic_gluten child294)
     (not_allergic_gluten child374)
     (waiting child294 table141)
     (waiting child374 table354)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child294)
     (served child374)
    )
  )
)
