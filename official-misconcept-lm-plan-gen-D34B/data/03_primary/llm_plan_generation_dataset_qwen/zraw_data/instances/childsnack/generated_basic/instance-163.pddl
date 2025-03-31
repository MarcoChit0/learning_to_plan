; child-snack task with 2 children and 0.33 gluten factor 
; constant factor of 1.7
; random seed: 847472

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child255 child351 - child
    bread360 bread12 - bread-portion
    content96 content349 - content-portion
    tray172 tray378 tray303 tray470 - tray
    table166 table268 table428 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray172 kitchen)
     (at tray378 kitchen)
     (at tray303 kitchen)
     (at tray470 kitchen)
     (at_kitchen_bread bread360)
     (at_kitchen_bread bread12)
     (at_kitchen_content content96)
     (at_kitchen_content content349)
     (not_allergic_gluten child255)
     (not_allergic_gluten child351)
     (waiting child255 table166)
     (waiting child351 table166)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child255)
     (served child351)
    )
  )
)
