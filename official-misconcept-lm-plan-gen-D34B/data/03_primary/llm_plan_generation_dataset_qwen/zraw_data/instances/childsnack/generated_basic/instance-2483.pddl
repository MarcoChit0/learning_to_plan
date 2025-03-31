; child-snack task with 2 children and 0.39 gluten factor 
; constant factor of 1.9
; random seed: 866126

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child84 child164 - child
    bread375 bread51 - bread-portion
    content265 content23 - content-portion
    tray220 tray140 - tray
    table408 table141 table80 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray220 kitchen)
     (at tray140 kitchen)
     (at_kitchen_bread bread375)
     (at_kitchen_bread bread51)
     (at_kitchen_content content265)
     (at_kitchen_content content23)
     (not_allergic_gluten child164)
     (not_allergic_gluten child84)
     (waiting child84 table141)
     (waiting child164 table408)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child84)
     (served child164)
    )
  )
)
