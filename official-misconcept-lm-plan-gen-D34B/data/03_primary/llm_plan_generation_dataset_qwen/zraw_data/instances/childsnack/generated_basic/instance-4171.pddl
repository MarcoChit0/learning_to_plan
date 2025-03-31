; child-snack task with 2 children and 0.28 gluten factor 
; constant factor of 1.7
; random seed: 763152

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child184 child428 - child
    bread277 bread253 - bread-portion
    content92 content280 - content-portion
    tray168 - tray
    table33 table207 table292 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray168 kitchen)
     (at_kitchen_bread bread277)
     (at_kitchen_bread bread253)
     (at_kitchen_content content92)
     (at_kitchen_content content280)
     (not_allergic_gluten child428)
     (not_allergic_gluten child184)
     (waiting child184 table33)
     (waiting child428 table33)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child184)
     (served child428)
    )
  )
)
