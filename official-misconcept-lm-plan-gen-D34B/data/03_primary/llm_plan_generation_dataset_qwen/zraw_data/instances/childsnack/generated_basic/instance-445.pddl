; child-snack task with 3 children and 0.32 gluten factor 
; constant factor of 1.9
; random seed: 709392

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child158 child460 child31 - child
    bread229 bread162 bread307 - bread-portion
    content114 content102 content471 - content-portion
    tray219 - tray
    table429 table397 table409 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray219 kitchen)
     (at_kitchen_bread bread229)
     (at_kitchen_bread bread162)
     (at_kitchen_bread bread307)
     (at_kitchen_content content114)
     (at_kitchen_content content102)
     (at_kitchen_content content471)
     (not_allergic_gluten child460)
     (not_allergic_gluten child31)
     (not_allergic_gluten child158)
     (waiting child158 table429)
     (waiting child460 table429)
     (waiting child31 table429)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child158)
     (served child460)
     (served child31)
    )
  )
)
