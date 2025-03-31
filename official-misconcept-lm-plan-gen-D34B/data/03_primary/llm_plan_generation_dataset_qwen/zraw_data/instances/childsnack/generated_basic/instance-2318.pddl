; child-snack task with 1 children and 0.7 gluten factor 
; constant factor of 1.3
; random seed: 877661

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child96 - child
    bread170 - bread-portion
    content435 - content-portion
    tray85 - tray
    table349 table70 table322 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray85 kitchen)
     (at_kitchen_bread bread170)
     (at_kitchen_content content435)
     (not_allergic_gluten child96)
     (waiting child96 table322)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child96)
    )
  )
)
