; child-snack task with 1 children and 0.77 gluten factor 
; constant factor of 1.9
; random seed: 278526

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child301 - child
    bread95 - bread-portion
    content337 - content-portion
    tray341 tray206 tray272 tray370 - tray
    table140 table58 table193 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray341 kitchen)
     (at tray206 kitchen)
     (at tray272 kitchen)
     (at tray370 kitchen)
     (at_kitchen_bread bread95)
     (at_kitchen_content content337)
     (not_allergic_gluten child301)
     (waiting child301 table193)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child301)
    )
  )
)
