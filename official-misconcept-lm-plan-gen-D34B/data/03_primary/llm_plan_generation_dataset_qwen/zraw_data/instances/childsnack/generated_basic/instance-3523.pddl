; child-snack task with 2 children and 0.42 gluten factor 
; constant factor of 1.3
; random seed: 327742

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child403 child118 - child
    bread456 bread92 - bread-portion
    content259 content363 - content-portion
    tray384 tray180 - tray
    table332 table415 table295 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray384 kitchen)
     (at tray180 kitchen)
     (at_kitchen_bread bread456)
     (at_kitchen_bread bread92)
     (at_kitchen_content content259)
     (at_kitchen_content content363)
     (not_allergic_gluten child403)
     (not_allergic_gluten child118)
     (waiting child403 table295)
     (waiting child118 table295)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child403)
     (served child118)
    )
  )
)
