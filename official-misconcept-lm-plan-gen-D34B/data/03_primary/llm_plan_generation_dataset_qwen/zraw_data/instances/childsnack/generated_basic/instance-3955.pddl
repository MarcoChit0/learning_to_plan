; child-snack task with 2 children and 0.78 gluten factor 
; constant factor of 1.9
; random seed: 21044

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child151 child183 - child
    bread463 bread184 - bread-portion
    content124 content139 - content-portion
    tray408 - tray
    table337 table103 table44 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray408 kitchen)
     (at_kitchen_bread bread463)
     (at_kitchen_bread bread184)
     (at_kitchen_content content124)
     (at_kitchen_content content139)
     (no_gluten_bread bread463)
     (no_gluten_content content124)
     (allergic_gluten child151)
     (not_allergic_gluten child183)
     (waiting child151 table103)
     (waiting child183 table337)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child151)
     (served child183)
    )
  )
)
