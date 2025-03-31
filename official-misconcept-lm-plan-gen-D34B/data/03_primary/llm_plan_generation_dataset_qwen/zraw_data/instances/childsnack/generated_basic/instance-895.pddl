; child-snack task with 3 children and 0.31 gluten factor 
; constant factor of 1.5
; random seed: 879546

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child129 child339 child448 - child
    bread369 bread306 bread278 - bread-portion
    content313 content346 content280 - content-portion
    tray291 - tray
    table339 table283 table10 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 - sandwich
  )
  (:init
     (at tray291 kitchen)
     (at_kitchen_bread bread369)
     (at_kitchen_bread bread306)
     (at_kitchen_bread bread278)
     (at_kitchen_content content313)
     (at_kitchen_content content346)
     (at_kitchen_content content280)
     (not_allergic_gluten child339)
     (not_allergic_gluten child448)
     (not_allergic_gluten child129)
     (waiting child129 table339)
     (waiting child339 table339)
     (waiting child448 table10)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
  )
  (:goal
    (and
     (served child129)
     (served child339)
     (served child448)
    )
  )
)
