; child-snack task with 2 children and 0.78 gluten factor 
; constant factor of 1.5
; random seed: 804007

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child480 child221 - child
    bread299 bread412 - bread-portion
    content68 content444 - content-portion
    tray68 tray168 tray458 tray266 - tray
    table361 table469 table279 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray68 kitchen)
     (at tray168 kitchen)
     (at tray458 kitchen)
     (at tray266 kitchen)
     (at_kitchen_bread bread299)
     (at_kitchen_bread bread412)
     (at_kitchen_content content68)
     (at_kitchen_content content444)
     (no_gluten_bread bread412)
     (no_gluten_content content68)
     (allergic_gluten child221)
     (not_allergic_gluten child480)
     (waiting child480 table279)
     (waiting child221 table469)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child480)
     (served child221)
    )
  )
)
