; child-snack task with 2 children and 0.42 gluten factor 
; constant factor of 1.3
; random seed: 762744

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child16 child249 - child
    bread400 bread319 - bread-portion
    content481 content142 - content-portion
    tray141 tray377 - tray
    table369 table59 table398 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray141 kitchen)
     (at tray377 kitchen)
     (at_kitchen_bread bread400)
     (at_kitchen_bread bread319)
     (at_kitchen_content content481)
     (at_kitchen_content content142)
     (not_allergic_gluten child16)
     (not_allergic_gluten child249)
     (waiting child16 table398)
     (waiting child249 table369)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child16)
     (served child249)
    )
  )
)
