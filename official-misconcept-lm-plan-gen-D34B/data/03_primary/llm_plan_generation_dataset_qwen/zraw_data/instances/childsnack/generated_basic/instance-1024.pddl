; child-snack task with 3 children and 0.3 gluten factor 
; constant factor of 1.3
; random seed: 993290

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child116 child130 child74 - child
    bread282 bread272 bread174 - bread-portion
    content99 content381 content245 - content-portion
    tray368 - tray
    table285 table141 table450 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray368 kitchen)
     (at_kitchen_bread bread282)
     (at_kitchen_bread bread272)
     (at_kitchen_bread bread174)
     (at_kitchen_content content99)
     (at_kitchen_content content381)
     (at_kitchen_content content245)
     (not_allergic_gluten child130)
     (not_allergic_gluten child116)
     (not_allergic_gluten child74)
     (waiting child116 table450)
     (waiting child130 table141)
     (waiting child74 table285)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child116)
     (served child130)
     (served child74)
    )
  )
)
