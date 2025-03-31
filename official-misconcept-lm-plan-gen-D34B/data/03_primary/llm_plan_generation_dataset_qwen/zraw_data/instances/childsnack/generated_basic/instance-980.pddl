; child-snack task with 3 children and 0.3 gluten factor 
; constant factor of 1.3
; random seed: 268994

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child25 child218 child361 - child
    bread53 bread95 bread226 - bread-portion
    content463 content392 content189 - content-portion
    tray426 - tray
    table205 table383 table343 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray426 kitchen)
     (at_kitchen_bread bread53)
     (at_kitchen_bread bread95)
     (at_kitchen_bread bread226)
     (at_kitchen_content content463)
     (at_kitchen_content content392)
     (at_kitchen_content content189)
     (not_allergic_gluten child218)
     (not_allergic_gluten child25)
     (not_allergic_gluten child361)
     (waiting child25 table205)
     (waiting child218 table383)
     (waiting child361 table205)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child25)
     (served child218)
     (served child361)
    )
  )
)
