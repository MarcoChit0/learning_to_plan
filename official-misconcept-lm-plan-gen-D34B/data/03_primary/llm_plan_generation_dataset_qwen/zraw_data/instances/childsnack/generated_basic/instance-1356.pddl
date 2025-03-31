; child-snack task with 2 children and 0.27 gluten factor 
; constant factor of 1.7
; random seed: 334058

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child26 child392 - child
    bread297 bread378 - bread-portion
    content38 content220 - content-portion
    tray338 tray150 tray117 - tray
    table433 table490 table355 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray338 kitchen)
     (at tray150 kitchen)
     (at tray117 kitchen)
     (at_kitchen_bread bread297)
     (at_kitchen_bread bread378)
     (at_kitchen_content content38)
     (at_kitchen_content content220)
     (not_allergic_gluten child26)
     (not_allergic_gluten child392)
     (waiting child26 table433)
     (waiting child392 table355)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child26)
     (served child392)
    )
  )
)
