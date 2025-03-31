; child-snack task with 2 children and 0.38 gluten factor 
; constant factor of 1.9
; random seed: 833358

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child79 child71 - child
    bread406 bread341 - bread-portion
    content363 content241 - content-portion
    tray66 - tray
    table224 table247 table324 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray66 kitchen)
     (at_kitchen_bread bread406)
     (at_kitchen_bread bread341)
     (at_kitchen_content content363)
     (at_kitchen_content content241)
     (not_allergic_gluten child79)
     (not_allergic_gluten child71)
     (waiting child79 table224)
     (waiting child71 table247)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child79)
     (served child71)
    )
  )
)
