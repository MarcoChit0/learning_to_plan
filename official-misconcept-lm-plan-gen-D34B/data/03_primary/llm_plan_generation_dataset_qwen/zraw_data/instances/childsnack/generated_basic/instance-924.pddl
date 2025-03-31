; child-snack task with 3 children and 0.31 gluten factor 
; constant factor of 1.5
; random seed: 736559

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child462 child496 child271 - child
    bread295 bread364 bread430 - bread-portion
    content300 content374 content198 - content-portion
    tray337 - tray
    table483 table235 table84 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 - sandwich
  )
  (:init
     (at tray337 kitchen)
     (at_kitchen_bread bread295)
     (at_kitchen_bread bread364)
     (at_kitchen_bread bread430)
     (at_kitchen_content content300)
     (at_kitchen_content content374)
     (at_kitchen_content content198)
     (not_allergic_gluten child496)
     (not_allergic_gluten child462)
     (not_allergic_gluten child271)
     (waiting child462 table235)
     (waiting child496 table84)
     (waiting child271 table235)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
  )
  (:goal
    (and
     (served child462)
     (served child496)
     (served child271)
    )
  )
)
