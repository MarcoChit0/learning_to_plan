; child-snack task with 3 children and 0.32 gluten factor 
; constant factor of 1.9
; random seed: 147554

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child371 child223 child244 - child
    bread381 bread276 bread179 - bread-portion
    content239 content473 content16 - content-portion
    tray203 - tray
    table474 table197 table14 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray203 kitchen)
     (at_kitchen_bread bread381)
     (at_kitchen_bread bread276)
     (at_kitchen_bread bread179)
     (at_kitchen_content content239)
     (at_kitchen_content content473)
     (at_kitchen_content content16)
     (not_allergic_gluten child244)
     (not_allergic_gluten child223)
     (not_allergic_gluten child371)
     (waiting child371 table14)
     (waiting child223 table197)
     (waiting child244 table14)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child371)
     (served child223)
     (served child244)
    )
  )
)
