; child-snack task with 3 children and 0.22 gluten factor 
; constant factor of 1.9
; random seed: 686914

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child191 child105 child188 - child
    bread135 bread113 bread182 - bread-portion
    content300 content88 content30 - content-portion
    tray174 - tray
    table76 table286 table159 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray174 kitchen)
     (at_kitchen_bread bread135)
     (at_kitchen_bread bread113)
     (at_kitchen_bread bread182)
     (at_kitchen_content content300)
     (at_kitchen_content content88)
     (at_kitchen_content content30)
     (not_allergic_gluten child191)
     (not_allergic_gluten child105)
     (not_allergic_gluten child188)
     (waiting child191 table286)
     (waiting child105 table159)
     (waiting child188 table286)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child191)
     (served child105)
     (served child188)
    )
  )
)
