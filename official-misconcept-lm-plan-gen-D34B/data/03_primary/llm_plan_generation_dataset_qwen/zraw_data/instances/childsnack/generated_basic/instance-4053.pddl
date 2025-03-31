; child-snack task with 3 children and 0.22 gluten factor 
; constant factor of 1.9
; random seed: 972911

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child204 child85 child213 - child
    bread162 bread158 bread268 - bread-portion
    content412 content376 content324 - content-portion
    tray443 - tray
    table269 table488 table119 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray443 kitchen)
     (at_kitchen_bread bread162)
     (at_kitchen_bread bread158)
     (at_kitchen_bread bread268)
     (at_kitchen_content content412)
     (at_kitchen_content content376)
     (at_kitchen_content content324)
     (not_allergic_gluten child85)
     (not_allergic_gluten child204)
     (not_allergic_gluten child213)
     (waiting child204 table119)
     (waiting child85 table119)
     (waiting child213 table488)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child204)
     (served child85)
     (served child213)
    )
  )
)
