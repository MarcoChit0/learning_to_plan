; child-snack task with 3 children and 0.32 gluten factor 
; constant factor of 1.9
; random seed: 236377

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child141 child237 child159 - child
    bread208 bread256 bread29 - bread-portion
    content425 content453 content20 - content-portion
    tray70 - tray
    table381 table467 table191 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray70 kitchen)
     (at_kitchen_bread bread208)
     (at_kitchen_bread bread256)
     (at_kitchen_bread bread29)
     (at_kitchen_content content425)
     (at_kitchen_content content453)
     (at_kitchen_content content20)
     (not_allergic_gluten child141)
     (not_allergic_gluten child159)
     (not_allergic_gluten child237)
     (waiting child141 table381)
     (waiting child237 table191)
     (waiting child159 table191)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child141)
     (served child237)
     (served child159)
    )
  )
)
