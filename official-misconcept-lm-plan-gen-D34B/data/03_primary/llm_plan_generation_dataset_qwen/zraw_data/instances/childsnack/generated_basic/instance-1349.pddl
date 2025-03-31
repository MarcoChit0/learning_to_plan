; child-snack task with 2 children and 0.27 gluten factor 
; constant factor of 1.7
; random seed: 402651

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child351 child119 - child
    bread278 bread337 - bread-portion
    content298 content129 - content-portion
    tray35 tray347 tray384 - tray
    table465 table469 table417 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray35 kitchen)
     (at tray347 kitchen)
     (at tray384 kitchen)
     (at_kitchen_bread bread278)
     (at_kitchen_bread bread337)
     (at_kitchen_content content298)
     (at_kitchen_content content129)
     (not_allergic_gluten child351)
     (not_allergic_gluten child119)
     (waiting child351 table469)
     (waiting child119 table417)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child351)
     (served child119)
    )
  )
)
