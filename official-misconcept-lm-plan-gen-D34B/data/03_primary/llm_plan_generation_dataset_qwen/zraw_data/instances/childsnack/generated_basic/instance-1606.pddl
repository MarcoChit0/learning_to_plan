; child-snack task with 1 children and 0.71 gluten factor 
; constant factor of 1.3
; random seed: 157716

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child126 - child
    bread284 - bread-portion
    content241 - content-portion
    tray373 tray109 tray118 tray317 - tray
    table332 table92 table498 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray373 kitchen)
     (at tray109 kitchen)
     (at tray118 kitchen)
     (at tray317 kitchen)
     (at_kitchen_bread bread284)
     (at_kitchen_content content241)
     (not_allergic_gluten child126)
     (waiting child126 table498)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child126)
    )
  )
)
