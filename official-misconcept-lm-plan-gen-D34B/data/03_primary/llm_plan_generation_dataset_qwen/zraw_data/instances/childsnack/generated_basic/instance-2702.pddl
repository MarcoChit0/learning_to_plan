; child-snack task with 3 children and 0.23 gluten factor 
; constant factor of 1.9
; random seed: 50655

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child94 child208 child194 - child
    bread72 bread247 bread362 - bread-portion
    content60 content180 content443 - content-portion
    tray176 tray370 tray136 - tray
    table147 table464 table146 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray176 kitchen)
     (at tray370 kitchen)
     (at tray136 kitchen)
     (at_kitchen_bread bread72)
     (at_kitchen_bread bread247)
     (at_kitchen_bread bread362)
     (at_kitchen_content content60)
     (at_kitchen_content content180)
     (at_kitchen_content content443)
     (not_allergic_gluten child208)
     (not_allergic_gluten child194)
     (not_allergic_gluten child94)
     (waiting child94 table464)
     (waiting child208 table146)
     (waiting child194 table147)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child94)
     (served child208)
     (served child194)
    )
  )
)
