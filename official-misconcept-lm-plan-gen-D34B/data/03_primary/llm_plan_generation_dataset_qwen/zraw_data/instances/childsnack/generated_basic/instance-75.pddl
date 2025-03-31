; child-snack task with 2 children and 0.36 gluten factor 
; constant factor of 1.9
; random seed: 43208

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child300 child153 - child
    bread452 bread402 - bread-portion
    content225 content126 - content-portion
    tray130 tray175 - tray
    table467 table133 table466 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray130 kitchen)
     (at tray175 kitchen)
     (at_kitchen_bread bread452)
     (at_kitchen_bread bread402)
     (at_kitchen_content content225)
     (at_kitchen_content content126)
     (not_allergic_gluten child300)
     (not_allergic_gluten child153)
     (waiting child300 table466)
     (waiting child153 table466)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child300)
     (served child153)
    )
  )
)
