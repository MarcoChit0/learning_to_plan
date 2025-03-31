; child-snack task with 1 children and 0.4 gluten factor 
; constant factor of 1.5
; random seed: 66654

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child79 - child
    bread195 - bread-portion
    content238 - content-portion
    tray245 tray96 - tray
    table412 table141 table219 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray245 kitchen)
     (at tray96 kitchen)
     (at_kitchen_bread bread195)
     (at_kitchen_content content238)
     (not_allergic_gluten child79)
     (waiting child79 table141)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child79)
    )
  )
)
