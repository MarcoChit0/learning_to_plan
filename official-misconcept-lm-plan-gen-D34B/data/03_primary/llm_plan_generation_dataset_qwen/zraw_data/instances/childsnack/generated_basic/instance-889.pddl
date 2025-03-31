; child-snack task with 3 children and 0.31 gluten factor 
; constant factor of 1.5
; random seed: 727053

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child377 child452 child279 - child
    bread442 bread207 bread155 - bread-portion
    content28 content349 content209 - content-portion
    tray109 - tray
    table169 table58 table221 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 - sandwich
  )
  (:init
     (at tray109 kitchen)
     (at_kitchen_bread bread442)
     (at_kitchen_bread bread207)
     (at_kitchen_bread bread155)
     (at_kitchen_content content28)
     (at_kitchen_content content349)
     (at_kitchen_content content209)
     (not_allergic_gluten child452)
     (not_allergic_gluten child279)
     (not_allergic_gluten child377)
     (waiting child377 table58)
     (waiting child452 table58)
     (waiting child279 table221)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
  )
  (:goal
    (and
     (served child377)
     (served child452)
     (served child279)
    )
  )
)
