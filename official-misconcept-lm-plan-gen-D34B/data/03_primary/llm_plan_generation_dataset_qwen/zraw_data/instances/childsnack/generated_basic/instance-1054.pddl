; child-snack task with 2 children and 0.41 gluten factor 
; constant factor of 1.3
; random seed: 269592

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child323 child362 - child
    bread77 bread14 - bread-portion
    content415 content316 - content-portion
    tray277 tray424 tray382 - tray
    table13 table290 table436 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray277 kitchen)
     (at tray424 kitchen)
     (at tray382 kitchen)
     (at_kitchen_bread bread77)
     (at_kitchen_bread bread14)
     (at_kitchen_content content415)
     (at_kitchen_content content316)
     (not_allergic_gluten child323)
     (not_allergic_gluten child362)
     (waiting child323 table290)
     (waiting child362 table436)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child323)
     (served child362)
    )
  )
)
